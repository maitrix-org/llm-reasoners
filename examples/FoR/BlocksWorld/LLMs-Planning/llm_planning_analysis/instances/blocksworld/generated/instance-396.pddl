(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects i h a b)
(:init 
(handempty)
(ontable i)
(ontable h)
(ontable a)
(ontable b)
(clear i)
(clear h)
(clear a)
(clear b)
)
(:goal
(and
(on i h)
(on h a)
(on a b)
)))