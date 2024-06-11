(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b i f e)
(:init 
(handempty)
(ontable b)
(ontable i)
(ontable f)
(ontable e)
(clear b)
(clear i)
(clear f)
(clear e)
)
(:goal
(and
(on b i)
(on i f)
(on f e)
)))