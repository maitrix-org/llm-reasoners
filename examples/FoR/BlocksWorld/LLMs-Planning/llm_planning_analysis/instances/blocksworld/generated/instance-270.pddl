(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b i g a)
(:init 
(handempty)
(ontable b)
(ontable i)
(ontable g)
(ontable a)
(clear b)
(clear i)
(clear g)
(clear a)
)
(:goal
(and
(on b i)
(on i g)
(on g a)
)))