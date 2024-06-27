(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f i a)
(:init 
(handempty)
(ontable f)
(ontable i)
(ontable a)
(clear f)
(clear i)
(clear a)
)
(:goal
(and
(on f i)
(on i a)
)))