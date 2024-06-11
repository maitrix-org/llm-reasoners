(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f i c a)
(:init 
(handempty)
(ontable f)
(ontable i)
(ontable c)
(ontable a)
(clear f)
(clear i)
(clear c)
(clear a)
)
(:goal
(and
(on f i)
(on i c)
(on c a)
)))