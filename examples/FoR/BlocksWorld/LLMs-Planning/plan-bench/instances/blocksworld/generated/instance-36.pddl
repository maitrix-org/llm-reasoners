(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h i a f)
(:init 
(handempty)
(ontable h)
(ontable i)
(ontable a)
(ontable f)
(clear h)
(clear i)
(clear a)
(clear f)
)
(:goal
(and
(on h i)
(on i a)
(on a f)
)))