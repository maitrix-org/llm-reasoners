(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b i g)
(:init 
(handempty)
(ontable b)
(ontable i)
(ontable g)
(clear b)
(clear i)
(clear g)
)
(:goal
(and
(on b i)
(on i g)
)))