(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b j l i)
(:init 
(handempty)
(ontable b)
(ontable j)
(ontable l)
(ontable i)
(clear b)
(clear j)
(clear l)
(clear i)
)
(:goal
(and
(on b j)
(on j l)
(on l i)
)))
